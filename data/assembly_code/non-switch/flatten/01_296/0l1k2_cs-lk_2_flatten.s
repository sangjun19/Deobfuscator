	.file	"0l1k2_cs-lk_2_flatten.c"
	.text
	.globl	_TIG_IZ_zpkv_envp
	.bss
	.align 8
	.type	_TIG_IZ_zpkv_envp, @object
	.size	_TIG_IZ_zpkv_envp, 8
_TIG_IZ_zpkv_envp:
	.zero	8
	.globl	_TIG_IZ_zpkv_argc
	.align 4
	.type	_TIG_IZ_zpkv_argc, @object
	.size	_TIG_IZ_zpkv_argc, 4
_TIG_IZ_zpkv_argc:
	.zero	4
	.globl	_TIG_IZ_zpkv_argv
	.align 8
	.type	_TIG_IZ_zpkv_argv, @object
	.size	_TIG_IZ_zpkv_argv, 8
_TIG_IZ_zpkv_argv:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"Child process %d with PID: %d\n"
.LC1:
	.string	"Error!"
	.text
	.globl	main
	.type	main, @function
main:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	$0, _TIG_IZ_zpkv_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_zpkv_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_zpkv_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 103 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-zpkv--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_zpkv_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_zpkv_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_zpkv_envp(%rip)
	nop
	movq	$8, -8(%rbp)
.L26:
	cmpq	$14, -8(%rbp)
	ja	.L28
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L8(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L8(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L8:
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L28-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L28-.L8
	.long	.L28-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L28-.L8
	.long	.L28-.L8
	.long	.L7-.L8
	.text
.L14:
	call	getchar@PLT
	movq	$14, -8(%rbp)
	jmp	.L18
.L7:
	movl	$0, %eax
	jmp	.L27
.L12:
	movl	$0, -24(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L18
.L16:
	addl	$1, -24(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L18
.L9:
	cmpl	$5, -24(%rbp)
	jg	.L20
	movq	$5, -8(%rbp)
	jmp	.L18
.L20:
	movq	$4, -8(%rbp)
	jmp	.L18
.L11:
	call	getpid@PLT
	movl	%eax, -16(%rbp)
	movl	-24(%rbp), %eax
	leal	1(%rax), %ecx
	movl	-16(%rbp), %eax
	movl	%eax, %edx
	movl	%ecx, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	call	getchar@PLT
	movl	$0, %edi
	call	exit@PLT
.L13:
	call	fork@PLT
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	movl	%eax, -20(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L18
.L10:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$1, %edi
	call	exit@PLT
.L17:
	cmpl	$0, -20(%rbp)
	jne	.L22
	movq	$9, -8(%rbp)
	jmp	.L18
.L22:
	movq	$1, -8(%rbp)
	jmp	.L18
.L15:
	cmpl	$0, -20(%rbp)
	jns	.L24
	movq	$10, -8(%rbp)
	jmp	.L18
.L24:
	movq	$0, -8(%rbp)
	jmp	.L18
.L28:
	nop
.L18:
	jmp	.L26
.L27:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	main, .-main
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
