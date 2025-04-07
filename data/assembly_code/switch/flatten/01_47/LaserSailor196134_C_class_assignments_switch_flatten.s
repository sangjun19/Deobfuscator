	.file	"LaserSailor196134_C_class_assignments_switch_flatten.c"
	.text
	.globl	_TIG_IZ_BjtV_argc
	.bss
	.align 4
	.type	_TIG_IZ_BjtV_argc, @object
	.size	_TIG_IZ_BjtV_argc, 4
_TIG_IZ_BjtV_argc:
	.zero	4
	.globl	_TIG_IZ_BjtV_argv
	.align 8
	.type	_TIG_IZ_BjtV_argv, @object
	.size	_TIG_IZ_BjtV_argv, 8
_TIG_IZ_BjtV_argv:
	.zero	8
	.globl	_TIG_IZ_BjtV_envp
	.align 8
	.type	_TIG_IZ_BjtV_envp, @object
	.size	_TIG_IZ_BjtV_envp, 8
_TIG_IZ_BjtV_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"you inputted a \"b\" or a \"B\""
	.align 8
.LC1:
	.string	"you didn't input an \"a\" or a \"b\""
.LC2:
	.string	"you inputted an \"a\" or an \"A\""
.LC3:
	.string	"input a character:"
	.text
	.globl	main
	.type	main, @function
main:
.LFB6:
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
	movq	$0, _TIG_IZ_BjtV_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_BjtV_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_BjtV_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 145 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-BjtV--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_BjtV_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_BjtV_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_BjtV_envp(%rip)
	nop
	movq	$1, -8(%rbp)
.L20:
	cmpq	$8, -8(%rbp)
	ja	.L22
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
	.long	.L22-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L22-.L8
	.long	.L7-.L8
	.text
.L11:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$8, -8(%rbp)
	jmp	.L15
.L7:
	movl	$0, %eax
	jmp	.L21
.L14:
	movq	$5, -8(%rbp)
	jmp	.L15
.L12:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$8, -8(%rbp)
	jmp	.L15
.L9:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$8, -8(%rbp)
	jmp	.L15
.L10:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	call	getchar@PLT
	movl	%eax, -16(%rbp)
	movl	-16(%rbp), %eax
	movb	%al, -17(%rbp)
	movsbl	-17(%rbp), %eax
	movl	%eax, %edi
	call	toupper@PLT
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	movb	%al, -17(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L15
.L13:
	movsbl	-17(%rbp), %eax
	cmpl	$65, %eax
	je	.L17
	cmpl	$66, %eax
	jne	.L18
	movq	$4, -8(%rbp)
	jmp	.L19
.L17:
	movq	$6, -8(%rbp)
	jmp	.L19
.L18:
	movq	$3, -8(%rbp)
	nop
.L19:
	jmp	.L15
.L22:
	nop
.L15:
	jmp	.L20
.L21:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
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
