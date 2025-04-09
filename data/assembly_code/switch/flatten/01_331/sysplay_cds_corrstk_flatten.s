	.file	"sysplay_cds_corrstk_flatten.c"
	.text
	.globl	_TIG_IZ_fFwx_argc
	.bss
	.align 4
	.type	_TIG_IZ_fFwx_argc, @object
	.size	_TIG_IZ_fFwx_argc, 4
_TIG_IZ_fFwx_argc:
	.zero	4
	.globl	_TIG_IZ_fFwx_argv
	.align 8
	.type	_TIG_IZ_fFwx_argv, @object
	.size	_TIG_IZ_fFwx_argv, 8
_TIG_IZ_fFwx_argv:
	.zero	8
	.globl	_TIG_IZ_fFwx_envp
	.align 8
	.type	_TIG_IZ_fFwx_envp, @object
	.size	_TIG_IZ_fFwx_envp, 8
_TIG_IZ_fFwx_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Here now!"
.LC1:
	.string	"%c\n"
.LC2:
	.string	"%*c %[^,]"
.LC3:
	.string	"Here!"
.LC4:
	.string	"Enter- a,b,c,d,e,f,g,h :"
.LC5:
	.string	"%[^,]"
	.text
	.globl	main
	.type	main, @function
main:
.LFB4:
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
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_fFwx_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_fFwx_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_fFwx_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 124 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-fFwx--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_fFwx_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_fFwx_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_fFwx_envp(%rip)
	nop
	movq	$5, -16(%rbp)
.L22:
	cmpq	$12, -16(%rbp)
	ja	.L25
	movq	-16(%rbp), %rax
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
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L25-.L8
	.long	.L12-.L8
	.long	.L25-.L8
	.long	.L11-.L8
	.long	.L25-.L8
	.long	.L10-.L8
	.long	.L25-.L8
	.long	.L9-.L8
	.long	.L25-.L8
	.long	.L25-.L8
	.long	.L7-.L8
	.text
.L7:
	movzbl	-17(%rbp), %eax
	movsbl	%al, %eax
	cmpl	$97, %eax
	je	.L15
	cmpl	$98, %eax
	jne	.L16
	movq	$9, -16(%rbp)
	jmp	.L17
.L15:
	movq	$9, -16(%rbp)
	jmp	.L17
.L16:
	movq	$9, -16(%rbp)
	nop
.L17:
	jmp	.L18
.L13:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -16(%rbp)
	jmp	.L18
.L12:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L23
	jmp	.L24
.L9:
	movzbl	-17(%rbp), %eax
	movsbl	%al, %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-17(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movzbl	-17(%rbp), %eax
	movsbl	%al, %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$7, -16(%rbp)
	jmp	.L18
.L11:
	movq	$0, -16(%rbp)
	jmp	.L18
.L14:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-17(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$7, -16(%rbp)
	jmp	.L18
.L10:
	movzbl	-17(%rbp), %eax
	cmpb	$101, %al
	je	.L20
	movq	$12, -16(%rbp)
	jmp	.L18
.L20:
	movq	$1, -16(%rbp)
	jmp	.L18
.L25:
	nop
.L18:
	jmp	.L22
.L24:
	call	__stack_chk_fail@PLT
.L23:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
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
